# Final Handover V4

> **Generated**: 2026-06-25
> **Project**: Bicycle Dynamics Model System Identification and Evaluation
> **Scope**: Comprehensive correction, verification, and calibration (P0-P6)
> **Status**: 6/6 agents COMPLETE

---

## 1. Executive Summary

### 1.1 What Changed from V3

| Issue | V3 Status | V4 Status | Action |
|-------|-----------|-----------|--------|
| gp_ensemble_5_0 MAE=0 | DIAGNOSED | FIXED (code) | `range(max(1, dagger_rounds))` |
| 500=1000 same MAE | DOCUMENTED | CONFIRMED | NaN filtering, not a bug |
| Metric units mixed | DOCUMENTED | FIXED | NMAE/NRMSE computed |
| Historical 0.066 | TRACED | CONFIRMED | Mode C phi-only, unseeded |
| DAgger semantics | AUDITED (WRONG) | CORRECTED | Two-step GP trajectory, NOT Version L |
| Uncertainty calibration | BASIC_ONLY | FULL | Scalar/Per-state/Conformal + OOD |
| survival_steps bug | UNKNOWN | FOUND | Line 87 never updated after break |
| endpoint_error bug | UNKNOWN | FOUND | Line 79 averages across states |
| 0.252 horizon | WRONG (100 steps) | CORRECTED (500 steps) | 3 documents had wrong label |

### 1.2 Key Corrections

1. **DAgger semantics audit was WRONG**: The V3 audit claimed "pure local dynamics residual" but code trace proves Round 1+ uses two-step GP trajectory comparison: `r_k = (s_real[k+1] - GP(GP(s_model[k], u), u)) / delta_std`

2. **0.252 is 500-step MAE, not 100-step**: Three documents incorrectly labeled it as "100 steps"

3. **survival_steps bug**: `eval_modes_v2.py` line 87 sets `survival_steps = n_valid` before the loop and never updates it after divergence break

4. **endpoint_error bug**: `eval_metrics.py` line 79 uses `np.mean(abs_errors[-1])` which averages across states

---

## 2. Model Performance (Verified)

### 2.1 Mode B (Fixed Action Open-Loop) — Per-State MAE at 500 Steps

| Model | phi (rad) | delta (rad) | phi_dot (rad/s) | delta_dot (rad/s) | NMAE |
|-------|-----------|-------------|-----------------|-------------------|------|
| gp_standard | 3.337 | 2.042 | 12.016 | 31.201 | 21.95 |
| gp_ensemble_7_5 | **0.116** | **0.119** | **0.403** | **0.370** | **0.520** |
| gp_ensemble_5_3 | 0.146 | 0.113 | 0.415 | 0.559 | 0.622 |
| sindy_4d | 3.643 | 4.605 | 8.618 | 9.903 | 15.96 |
| linearized_model | 3.122 | 8.664 | 3.704 | 8.916 | 19.87 |
| gp_b4_sparse | 0.539 | 0.459 | 1.455 | 32.017 | 15.30 |
| gp_ensemble_5_0 | 0.000 | 0.000 | 0.000 | 0.000 | BUG |

### 2.2 Ablation Experiment (Agent A, 30000 training samples)

| Model | Description | Mode A 1000-step | Mode B 1000-step | Status |
|-------|-------------|-----------------|-----------------|--------|
| E0 | Pure GP | ~0 | 17.18 | Diverges (Mode B) |
| E1 | GP + offline NN | ~0 | 21.41 | Diverges (Mode B) |
| E2 | GP + NN + 3-round DAgger | 0.0068 | 0.297 | Stable |
| E3 | gp_ensemble_7_5 (7 models, 5 rounds) | 0.0079 | **0.264** | Stable, BEST |

**Key finding**: DAgger/residual training is essential for multi-step open-loop prediction. Pure GP diverges in Mode B despite being perfect in Mode A (teacher forcing).

### 2.3 NMAE (Normalized MAE) — Dimensionless, Physically Meaningful

| Model | 10 steps | 50 steps | 100 steps | 500 steps | 1000 steps | Status |
|-------|----------|----------|-----------|-----------|------------|--------|
| gp_standard | 0.000 | 0.001 | 0.005 | 21.95 | 31.57 | Diverges ~200 |
| gp_ensemble_7_5 | 0.238 | 0.443 | 0.473 | **0.520** | **0.531** | Stable |
| gp_ensemble_5_3 | 0.210 | 0.539 | 0.581 | 0.622 | 0.631 | Stable |
| sindy_4d | 0.040 | 0.120 | 0.812 | 15.96 | 15.96 | Diverges ~100 |
| linearized_model | 0.273 | 0.329 | 1.463 | 19.87 | 19.87 | Diverges ~80 |
| gp_b4_sparse | 0.321 | 1.392 | 2.731 | 15.30 | 15.30 | Diverges ~130 |
| gp_ensemble_5_0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | BUG |

### 2.4 Divergence Steps (Per-Model, 5 Seeds)

| Model | Mean Divergence Step | Notes |
|-------|---------------------|-------|
| gp_standard | ~200 | V3 claimed "500-1000" but actually ~200 |
| sindy_4d | ~85 | Consistent with V3 |
| linearized_model | ~66 | V3 claimed "100-500" but actually ~66 |
| gp_b4_sparse | ~105 | High variance (4-145) |
| gp_ensemble_7_5 | Never | Survived all 1000 steps |
| gp_ensemble_5_3 | Never | Survived all 1000 steps |

---

## 3. Bug Inventory

### 3.1 CRITICAL Bugs

| Bug | Location | Root Cause | Fix | Status |
|-----|----------|------------|-----|--------|
| gp_ensemble_5_0 MAE=0 | eval_models.py L305 | `range(0)` never executes | `range(max(1, dagger_rounds))` | FIXED (small numerical diff remains) |
| survival_steps wrong | eval_modes_v2.py L87 | Set before loop, never updated | `survival_steps = i + 1` after break | DIAGNOSED |
| endpoint_error wrong | eval_metrics.py L79 | Averages across states | Per-state version at L113 is correct | DIAGNOSED |

### 3.2 DESIGN Issues

| Issue | Description | Impact |
|-------|-------------|--------|
| Metric units mixed | Overall MAE mixes rad and rad/s | Physically meaningless |
| NaN filtering after divergence | 500=1000 same MAE | Misleading, need survival_steps |
| DAgger Round 0 vs 1+ inconsistency | Different residual formulas | NN trained on mixed objectives |

---

## 4. DAgger Semantics (Corrected)

### 4.1 Current Implementation (Confirmed by Code Trace)

**Round 0** (eval_models.py L288-292):
```
r_k = (F_real(s_expert, u) - GP(s_expert, u)) / delta_std
```
Pure Version L: local dynamics residual at expert states.

**Round 1+** (eval_models.py L352-379):
```
r_k = (s_real[k+1] - GP(GP(s_model[k], u), u)) / delta_std
```
Two-step GP trajectory comparison. NOT Version L, NOT standard Version T.

### 4.2 Version L vs Version T Comparison

| Metric | Version L | Version T | Winner |
|--------|-----------|-----------|--------|
| Mode A (Teacher Forcing) | **0.0025** | 0.0339 | L (13x better) |
| Mode B (Open-loop 100 steps) | 0.619 | **0.436** | T (30% better) |
| Mode D (Closed-loop LQR) | **0.0045** | 0.1389 | L (31x better) |
| Initial: large_phi | 0.659 | **0.398** | T (40% better) |
| OOD: extreme_phi | **0.0013** | 0.0274 | L (21x better) |

**Conclusion**: Version L is better for single-step accuracy and closed-loop control. Version T is better for multi-step open-loop robustness. The current implementation mixes both approaches across rounds.

### 4.3 Naming Recommendation

"DAgger" is misleading. Recommended: **"Residual Dynamics Ensemble" (RDE)** with RDE-L and RDE-T variants.

---

## 5. Uncertainty Calibration (Complete)

### 5.1 Data Split

| Split | Samples | Purpose |
|-------|---------|---------|
| Train | 2500 | Model training |
| Calibration | 1250 | Calibration parameter fitting |
| Test | 1250 | Final evaluation |

### 5.2 Pre-Calibration (Raw Uncertainty)

| State | Pearson r | Coverage_95 | Coverage_68 |
|-------|-----------|-------------|-------------|
| phi | 0.487 | 65.0% | 38.3% |
| delta | 0.613 | 60.5% | 37.4% |
| phi_dot | 0.685 | 42.4% | 24.9% |
| delta_dot | 0.228 | 86.8% | 58.9% |
| **Overall** | **0.540** | **48.8%** | **31.2%** |

### 5.3 Post-Calibration (Per-State Scaling, Test Set)

| State | Scale Factor | Coverage_95 | Coverage_68 |
|-------|-------------|-------------|-------------|
| phi | 8.37 | 99.7% | 92.5% |
| delta | 4.87 | 99.7% | 94.0% |
| phi_dot | 4.73 | 100% | 94.3% |
| delta_dot | 4.29 | 99.7% | 97.7% |
| **Overall** | — | **99.7%** | **93.0%** |

### 5.4 Mode-Isolated Calibration

| Mode | Raw Coverage_95 | Scalar Coverage_95 | Conformal Coverage_95 |
|------|----------------|-------------------|----------------------|
| A | 59.0% | 99.7% | 99.7% |
| B/C | 8.0% | 74.0% | 100% |
| D | 30.0% | 93.0% | 100% |

### 5.5 OOD Detection

| Metric | Value |
|--------|-------|
| AUROC | 0.795 |
| AUPRC | 0.833 |
| ID mean uncertainty | 0.082 |
| OOD mean uncertainty | 0.127 |
| OOD/ID ratio | 1.55x |

---

## 6. Historical Result Corrections

### 6.1 The 0.066/0.064 Mystery (Resolved)

| Value | Source | Mode | Metric | Steps | Seeds |
|-------|--------|------|--------|-------|-------|
| 0.066 | test_ensemble_optimization.py | C | phi-only endpoint error | 500 | unseeded |
| 0.064 | Same source, different run | C | phi-only endpoint error | 500 | unseeded |

**Conclusion**: These are Mode C phi-only endpoint errors at 500 steps, not comparable to full-trajectory MAE.

### 6.2 The 0.252 Contradiction (Resolved)

| Document | Claimed | Actual |
|----------|---------|--------|
| FINAL_HANDOVER_V3.md L78 | "0.252 at step 100" | **0.252 at step 500** |
| HISTORICAL_0064_0066_CODE_TRACE.md L13 | "0.252 at step 100" | **0.252 at step 500** |
| HISTORICAL_RESULT_FACTS.json | steps: 100 | **steps: 500** |
| FINAL_HANDOVER_V3.md L134 | "0.252 under 500-step column" | Correct |

**Conclusion**: 0.252 is the Mode B 500-step MAE for gp_ensemble_7_5. Three documents had the wrong horizon label.

---

## 7. 8D Route Recommendations

### 7.1 State Reclassification

**Current (8D)**: [e_y, e_psi, v, theta, theta_dot, kappa, delta, delta_dot]

**Recommended (7D + 2D input)**:
- States: [e_y, e_psi, v, theta, theta_dot, delta, delta_dot]
- Exogenous inputs: [kappa (curvature), a (acceleration)]

### 7.2 Data Requirements

| Requirement | Current | Needed |
|-------------|---------|--------|
| Samples | 2083 | >5000 (for 7D GP) |
| Parameter ratio | 440 params / 2083 samples | Need >10x data |

---

## 8. Recommended Configuration

### 8.1 Best Model: gp_ensemble_7_5

| Application | Model | Horizon | Expected NMAE |
|-------------|-------|---------|---------------|
| Short-term prediction | gp_standard | ≤50 steps | <0.001 |
| Multi-step planning | gp_ensemble_7_5 | 50-500 steps | 0.44-0.52 |
| Closed-loop control | gp_ensemble_7_5 | 1000+ steps | 0.53 |

### 8.2 Uncertainty Usage

| Scenario | Recommendation |
|----------|---------------|
| Risk assessment | Use per-state calibrated uncertainty |
| OOD detection | Ensemble std > 0.127 likely OOD |
| Mode-specific | Calibrate separately per evaluation mode |
| Decision making | Always apply calibration scaling before use |

---

## 9. Deliverables

### 9.1 continuation_stage_v4/

| Directory | Files | Status |
|-----------|-------|--------|
| agent_a_zero_dagger/ | fixed_eval_models.py, run_ablation_experiment.py, ZERO_DAGGER_ROOT_CAUSE.md, ZERO_DAGGER_FIXED_IMPLEMENTATION.md, results/ENSEMBLE_EQUIVALENCE_RESULTS.json, results/ENSEMBLE_ABLATION_RESULTS.json, OFFLINE_NN_BASELINE_RESULTS.json, results/SELF_CHECK.json | COMPLETE |
| agent_b_evaluation/ | 7 files: MODE_B_CALL_GRAPH.md, MODE_B_PROTOCOL_PROOF.md, DIVERGENCE_HANDLING_AUDIT.md, PER_STATE_METRICS.csv, NORMALIZED_METRICS.csv, RESULT_0252_HORIZON_TRACE.md, SELF_CHECK.json | COMPLETE |
| agent_c_history/ | 5 files: HISTORICAL_RESULT_CODE_TRACE.md, HISTORICAL_RESULT_FACTS.json, HISTORICAL_RESULT_TIMELINE.csv, RESULT_CONFLICT_FINAL_DECISION.md, SELF_CHECK.json | COMPLETE |
| agent_d_dagger/ | 9 files: CURRENT_DAGGER_EXACT_FORMULA.md, version_l_local_residual.py, version_t_trajectory_residual.py, LOCAL_VS_TRAJECTORY_RESULTS.json, DAGGER_NAMING_DECISION.md, SELF_CHECK.json, etc. | COMPLETE |
| agent_e_uncertainty/ | 8 files: CALIBRATION_SPLIT.json, RAW_UNCERTAINTY_AUDIT.md, SCALAR_SCALING_RESULTS.json, PER_STATE_SCALING_RESULTS.json, CONFORMAL_RESULTS.json, OOD_RESULTS.csv, POST_CALIBRATION_REPORT.md, SELF_CHECK.json | COMPLETE |
| agent_f_route8d/ | 6 files: ROUTE8D_STATE_RECLASSIFICATION.md, ROUTE8D_LAYERED_ARCHITECTURE.md, ROUTE8D_DATA_REQUIREMENTS.csv, ROUTE8D_MINIMUM_VALIDATION_PLAN.md, ROUTE8D_DECISION_FACTS.json, SELF_CHECK.json | COMPLETE |
| orchestration/ | CURRENT_TASK_SNAPSHOT_V4.json | COMPLETE |
| reports/ | FINAL_HANDOVER_V4.md, FINAL_FACTS_V4.json, FINAL_COMPLETION_MATRIX_V4.md | COMPLETE |

---

## 10. Remaining Work

| Item | Priority | Status |
|------|----------|--------|
| gp_ensemble_5_0 fix verification | HIGH | FIXED (3.11e-09 diff, rollout compounds) |
| survival_steps bug fix | MEDIUM | DIAGNOSED, needs code change |
| endpoint_error bug fix | LOW | DIAGNOSED, per-state version exists |
| Agent A ablation experiment | HIGH | COMPLETE (E0/E1/E2/E3 comparison done) |
| DAgger Round 0/1+ consistency | LOW | DOCUMENTED, no immediate fix |
| 8D route GP/NN model | LOW | NOT_STARTED, needs more data |

---

*Report generated: 2026-06-25*
*Agents: A(COMPLETE), B(COMPLETE), C(COMPLETE), D(COMPLETE), E(COMPLETE), F(COMPLETE)*
*Total experiment runtime: ~90 minutes*
