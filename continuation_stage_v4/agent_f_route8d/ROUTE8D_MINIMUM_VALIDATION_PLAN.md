# 8D Route Minimum Validation Plan

> **Generated**: 2026-06-25
> **Agent**: F (Route 8D Restructuring Planner)
> **Scope**: Structure decisions + minimum validation for the restructured 7D model

---

## 1. Current State Assessment

### 1.1 What Exists

| Component | Status | Evidence |
|-----------|--------|----------|
| 8D SINDy model (original) | EXISTS, low quality | `sindy_model.npz`, 55x8 coefficients |
| 8D SINDy model (improved) | EXISTS, requires 0.5x damping | `sindy_model_improved.npz`, 55x8 coefficients |
| 8D GP model | DOES NOT EXIST | All GP code is 4D-specific |
| 8D NN model | DOES NOT EXIST | `ResidualNet` hardcoded for 4D |
| 8D evaluation framework | DOES NOT EXIST | No 8D-specific eval code |
| 8D data | EXISTS, small | 2083 samples, ~10 episodes |

### 1.2 What Needs to Happen

| Phase | Task | Priority |
|-------|------|----------|
| Phase 1 | Structure Decision (this document) | DONE |
| Phase 2 | Code Restructuring (7D + layered) | P0 |
| Phase 3 | Data Collection (new episodes) | P1 |
| Phase 4 | SINDy Re-identification | P1 |
| Phase 5 | Single-step Validation | P2 |
| Phase 6 | Multi-step Rollout Validation | P2 |
| Phase 7 | GP/NN Model Training | P3 |

---

## 2. Phase 1: Structure Decision (COMPLETE)

### 2.1 Decisions Made

| Decision | Choice | Rationale |
|----------|--------|-----------|
| State dimension | **7D** | Remove k (exogenous); keep v (dynamic state) |
| Input dimension | **2D** | [tau, kappa] |
| Modeling approach | **Layered** (5 analytical + 2 SINDy) | Physical consistency + fewer parameters |
| k treatment | **Exogenous input** | CONFIRMED: constant per episode |
| v treatment | **Dynamic state** | Physical state; varies across episodes |
| Path kinematics | **Analytical** | Exact equations, no learning needed |
| Balance dynamics | **SINDy for theta_dot only** | theta is kinematic integration |
| Steering dynamics | **SINDy for delta_dot only** | delta is kinematic integration |

### 2.2 Confidence Levels

| Fact | Confidence | Basis |
|------|------------|-------|
| k is constant per episode | CONFIRMED | Code: `data_collector.py:128-135,209` |
| v is constant per episode | CONFIRMED | Code: `bicycle_dynamics.py:160` (dv/dt=0) |
| Path kinematics are exact | CONFIRMED | Geometric identity from bicycle model |
| theta integration is kinematic | CONFIRMED | SINDy identifies only theta_dot term for theta |
| delta integration is kinematic | CONFIRMED | SINDy identifies only delta_dot term for delta |
| 7D is better than 6D | INFERRED | v variation provides useful excitation |
| Layered is better than monolithic | INFERRED | 84% fewer parameters; physical consistency |
| 0.5x damping will be eliminated | UNKNOWN | Key validation target |

---

## 3. Phase 2: Code Restructuring (Planning Only)

### 3.1 Files to Modify (No modifications in this phase)

| File | Changes | Risk |
|------|---------|------|
| `data_collector.py` | Remove k from state; add kappa to inputs; update normalization | LOW |
| `sindy_identification.py` | Update state/input dimensions; add layered identification | MEDIUM |
| `world_model.py` | Update for 7D state + layered structure | MEDIUM |
| `sindy_env.py` | Update observation space; remove 0.5x damping | MEDIUM |
| `bicycle_dynamics.py` | NO CHANGES (physics engine) | NONE |

### 3.2 New Files to Create

| File | Purpose |
|------|---------|
| `sindy_layered.py` | Layered SINDy identification (theta_dot + delta_dot) |
| `eval_8d.py` | 8D multi-step evaluation framework |
| `collect_8d_data.py` | Targeted data collection script |

### 3.3 Restructured Data Format

```python
# Before (8D):
states_8d = [ey, epsi, v, theta, theta_dot, k, delta, delta_dot]  # (n, 8)
action = tau  # (n, 1)

# After (7D + 2D input):
states_7d = [ey, epsi, v, theta, theta_dot, delta, delta_dot]  # (n, 7)
inputs_2d = [tau, kappa]  # (n, 2)
```

---

## 4. Phase 3: Data Collection Plan

### 4.1 Current Data Analysis

| Metric | Value | Assessment |
|--------|-------|------------|
| Total samples | 2083 | MARGINAL for 55-term library (440 params) |
| Estimated episodes | ~10 | LOW (need 50+ for robust identification) |
| Speed range | [2.0, 6.0] m/s | GOOD |
| Speed coverage: [2,3) | 828 (39.8%) | OVER-represented |
| Speed coverage: [3,4) | 616 (29.6%) | Adequate |
| Speed coverage: [4,5) | 373 (17.9%) | UNDER-represented |
| Speed coverage: [5,6] | 266 (12.8%) | UNDER-represented |
| k range | [-0.294, 0.296] | GOOD |
| k: straight ([-0.02,0.02]) | 876 (42.1%) | OVER-represented |
| k: curved ([-0.1,0.1]) | 1605 (77.1%) | Adequate |
| k: sharp ([-0.3,0.3]) | 2083 (100%) | Adequate |

### 4.2 Data Gaps

| Gap | Severity | Fix |
|-----|----------|-----|
| Too few episodes (~10) | HIGH | Collect 50+ episodes per scenario |
| Over-representation of low-speed | MEDIUM | Use stratified speed sampling |
| Over-representation of straight | MEDIUM | Equal-weight scenario sampling |
| No time-varying curvature | LOW | Add trajectory-level curvature variation |
| No excitation near boundaries | MEDIUM | Extend initial state ranges |

### 4.3 Recommended Data Collection

For the layered model (72 parameters), the minimum data requirement is:

```
N_min = 10 * n_params_per_layer = 10 * 36 = 360 samples per SINDy model
N_target = 50 * 36 = 1800 samples per SINDy model
N_total = 1800 * 2 (two layers) = 3600 samples
```

**Recommended collection**:
- 50 episodes per scenario (straight, curved, sharp) = 150 episodes total
- 8 seconds per episode at 30 Hz = 240 samples/episode
- Expected: ~150 * 200 (after fall filtering) = 30,000 samples
- This provides 250x over-determination for 72 parameters

### 4.4 Excitation Requirements

| Excitation Type | Purpose | Current Coverage | Needed |
|-----------------|---------|------------------|--------|
| Random walk on tau | Explore steering space | GOOD (PD + noise) | Maintain |
| Speed variation | Explore v-dependent coupling | PARTIAL (2-6 m/s) | Improve uniformity |
| Curvature variation | Explore kappa-dependent dynamics | GOOD (3 scenarios) | Maintain |
| Near-fall trajectories | Explore nonlinear roll dynamics | LIMITED (filtered at 1.0 rad) | Extend to 0.5 rad |
| High-delta trajectories | Explore steering nonlinearity | GOOD | Maintain |
| Combined excitation | Cross-layer coupling | LIMITED | Add simultaneous perturbations |

---

## 5. Phase 4: SINDy Re-identification Plan

### 5.1 Strategy: Layer-by-Layer

**Layer 2 (Balance)**:
```
Features: [theta, theta_dot, delta, delta_dot, v, kappa, tau]  (7 features)
Library: degree-2 polynomial → 36 terms
Output: theta_dot (1D)
Target: SINDy identification of f_balance
```

**Layer 3 (Steering)**:
```
Features: [delta, delta_dot, theta, theta_dot, v, kappa, tau]  (7 features)
Library: degree-2 polynomial → 36 terms
Output: delta_dot (1D)
Target: SINDy identification of f_steer
```

### 5.2 Hyperparameter Sweep

| Parameter | Range | Notes |
|-----------|-------|-------|
| threshold | [0.001, 0.005, 0.01, 0.02, 0.05, 0.1] | Sparsity threshold |
| alpha | [0.0, 0.01, 0.05, 0.1, 0.5] | L2 regularization |
| degree | [2, 3] | Polynomial degree |
| max_iter | [50] | STLSQ iterations |

### 5.3 Expected Outcome

Based on the existing SINDy model and physics:
- theta_dot: 3-5 active terms (theta, v*delta, v*theta_dot, theta_dot)
- delta_dot: 8-15 active terms (tau, delta, delta_dot, theta, v*delta, v*delta_dot, theta*delta, etc.)

---

## 6. Phase 5: Minimum Validation Protocol

### 6.1 Single-Step Prediction

| Test | Metric | Threshold | Status |
|------|--------|-----------|--------|
| theta integration accuracy | RMSE of (theta_true - theta_predicted) | < 0.001 rad | TO DO |
| delta integration accuracy | RMSE of (delta_true - delta_predicted) | < 0.001 rad | TO DO |
| theta_dot prediction R^2 | R^2 score | > 0.95 | TO DO |
| delta_dot prediction R^2 | R^2 score | > 0.90 | TO DO |
| Overall single-step RMSE | RMSE across all 7 states | < 0.01 (normalized) | TO DO |

### 6.2 Multi-Step Rollout

| Steps | Acceptable MAE | Purpose |
|-------|---------------|---------|
| 1 | < 0.001 rad | Single-step accuracy |
| 5 | < 0.005 rad | Short-term dynamics |
| 10 | < 0.01 rad | Medium-term stability |
| 20 | < 0.02 rad | Divergence onset |
| 50 | < 0.05 rad | Long-term behavior |
| 100 | < 0.10 rad | Extended prediction |
| 200 | Report survival | Maximum horizon |

### 6.3 Ablation Tests

| Test | Procedure | Expected Result |
|------|-----------|-----------------|
| No damping factor | Remove 0.5x multiplier in sindy_env.py | Model should be stable without damping |
| k as state vs input | Compare 8D (k-in-state) vs 7D (k-as-input) | 7D should have lower RMSE |
| Monolithic vs layered | Compare single 55x7 SINDy vs two 36x1 SINDy | Layered should have comparable or better RMSE |
| Kinematic vs learned | Compare analytical path kinematics vs SINDy-learned | Analytical should be exact |

### 6.4 Validation Against Ground Truth

The ground truth is the `bicycle_dynamics.py` simulator with known parameters. Validation:

1. **Open-loop**: Simulate 100 episodes with random initial states and actions; compare model rollout vs simulator
2. **Closed-loop**: Run LQR controller with the model; compare trajectory quality vs simulator
3. **Stability**: Check if the model remains stable for 500+ steps without damping

---

## 7. Phase 6: Multi-Step Rollout Validation (Detailed)

### 7.1 Evaluation Script Design

```python
def evaluate_8d_rollout(model, data, step_counts=[1, 5, 10, 20, 50, 100, 200]):
    """
    Evaluate model rollout at multiple horizons.

    For each initial condition in data:
        1. Run model for max(step_counts) steps using true actions
        2. Compute MAE at each checkpoint
        3. Report per-state and aggregate statistics

    Returns:
        results: dict[step_count] -> {mae_per_state, mae_total, survival_rate}
    """
```

### 7.2 Metrics

| Metric | Definition | Purpose |
|--------|-----------|---------|
| MAE per state | Mean absolute error at each step | Per-variable accuracy |
| MAE total | Mean across all states | Overall accuracy |
| NMAE | Normalized MAE (divided by state range) | Cross-variable comparison |
| Survival rate | Fraction of rollouts that don't diverge | Stability |
| Divergence step | Step at which MAE > threshold | Onset of divergence |
| Pearson r | Correlation between predicted and true states | Trend accuracy |

### 7.3 Divergence Criteria

A rollout is considered "diverged" if:
- Any state exceeds 10x its typical range (e.g., |theta| > 1.0 rad)
- Any state becomes NaN
- MAE exceeds 1.0 (normalized)

---

## 8. Phase 7: GP/NN Model Training (Future)

### 8.1 Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Training data | 3000 samples | 30,000 samples |
| Speed coverage | Uniform [2,6] | Uniform + perturbation |
| Curvature coverage | 3 scenarios | 5+ scenarios |
| GP kernel | RBF | RBF + periodic |
| NN architecture | ResidualNet(7, 1) | ResidualNet(7, 1) + ensemble |
| Training epochs | 50 | 100 |
| Validation split | 20% | 20% with temporal separation |

### 8.2 Port Strategy

The 4D GP/NN code can be ported to 7D by:
1. Changing `state_dim` from 4 to 7
2. Updating normalization constants
3. Adding kappa as an input feature
4. Updating evaluation scripts

---

## 9. Execution Roadmap

| Phase | Duration | Dependencies | Deliverable |
|-------|----------|--------------|-------------|
| P1: Structure Decision | DONE | None | This document |
| P2: Code Restructuring | 1-2 hours | P1 | Modified data_collector.py, sindy_layered.py |
| P3: Data Collection | 5-10 minutes (simulation) | P2 | bicycle_data_7d.npz |
| P4: SINDy Re-identification | 5-10 minutes | P3 | sindy_layered_model.npz |
| P5: Single-step Validation | 5 minutes | P4 | Validation results |
| P6: Multi-step Rollout | 10-15 minutes | P4 | Rollout results |
| P7: GP/NN Training | 30-60 minutes | P3 | gp_7d.npz, nn_7d.pt |

**Total estimated time for P2-P6: 30-60 minutes**

---

## 10. Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Layered model doesn't improve over monolithic | LOW | HIGH | Fall back to monolithic 7D SINDy |
| Data collection produces too few samples | LOW | MEDIUM | Use existing data with k removed |
| theta_dot SINDy overfits | MEDIUM | MEDIUM | Use higher threshold, more regularization |
| delta_dot SINDy has too many terms | MEDIUM | LOW | Accept 15-20 terms; it is the complex equation |
| 0.5x damping is still needed | LOW | HIGH | Indicates fundamental model inadequacy |
| Kinematic equations don't match data | VERY LOW | HIGH | Bug in data generation |

---

*Validation plan created: 2026-06-25*
*Agent F (Route 8D Restructuring Planner)*
