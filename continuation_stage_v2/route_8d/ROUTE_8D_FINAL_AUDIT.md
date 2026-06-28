# 8D Route Final Audit Report

> **Generated**: 2026-06-25
> **Project**: sindy_bicycle (D:\系统辨识作业\sindy_bicycle)
> **Audit Scope**: 8D path-tracking route -- data, models, state properties, evaluation
> **Evidence Levels**: CONFIRMED (code-verified), INFERRED (reasoned), UNKNOWN (unconfirmed)

---

## 1. State Definition

8D state vector: **x = [ey, epsi, v, theta, theta_dot, k, delta, delta_dot]**

| Index | Name | Unit | Physical Meaning |
|-------|------|------|------------------|
| 0 | ey | m | Lateral error from reference path |
| 1 | epsi | rad | Heading error relative to reference path |
| 2 | v | m/s | Longitudinal speed |
| 3 | theta | rad | Roll (lean) angle |
| 4 | theta_dot | rad/s | Roll angular velocity |
| 5 | k | 1/m | Reference path curvature |
| 6 | delta | rad | Steering angle |
| 7 | delta_dot | rad/s | Steering angular velocity |

**Control input**: Steering torque (N*m), 1D, range [-10, 10]

**Evidence**: CONFIRMED by `bicycle_dynamics.py:7-18`, `data_collector.py:28`

---

## 2. Data Audit

### 2.1 Data Files

| File | Samples | State Shape | Action Shape | dt |
|------|---------|-------------|--------------|----|
| bicycle_data.npz | 468 | (468, 8) | (468, 1) | 1/30 s |
| bicycle_data_improved.npz | 2,083 | (2083, 8) | (2083, 1) | 1/30 s |

**Evidence**: CONFIRMED by `reports/NEXT_STAGE_BASELINE_AUDIT.md:246-247` and code inspection

### 2.2 Data Generation Method

- **Simulator**: Simplified Whipple bicycle model (`bicycle_dynamics.py:whiple_eom()`)
- **Integration**: RK4 with dt=1/30 s (30 Hz)
- **Full state**: 10D `[x, y, psi, v, theta, theta_dot, delta, delta_dot, phi_f, omega_f]`
- **Conversion**: `full_state_to_path_tracking()` projects to 8D path-tracking frame
- **Controller**: PD balance controller + random exploration perturbations
  - PD gains: `Kp_roll=8.0, Kd_roll=2.0, Kp_delta=-1.5`
  - Exploration noise: exponential inter-arrival, uniform [-2, 2], scenario-weighted

### 2.3 Scenario Types

| Scenario | k_ref range | Exploration weight | Purpose |
|----------|-------------|-------------------|---------|
| straight | [-0.02, 0.02] | 0.3 | Baseline straight-line tracking |
| curved | [-0.1, 0.1] | 0.5 | Moderate curve tracking |
| sharp | [-0.3, 0.3] | 0.8 | Aggressive turns |

### 2.4 Initial State Distribution

| Variable | Range | Unit |
|----------|-------|------|
| v | [2.0, 6.0] | m/s |
| y | [-1.0, 1.0] | m |
| psi | [-0.2, 0.2] | rad |
| theta | [-0.15, 0.15] | rad |
| theta_dot | [-0.5, 0.5] | rad/s |
| delta | [-0.2, 0.2] | rad |
| delta_dot | [-0.5, 0.5] | rad/s |

### 2.5 Termination / Fall Conditions

- **Fall filter**: `|theta| > 1.0 rad` -- samples with roll angle exceeding 1.0 rad are removed
- **Numerical guard**: `|any state| > 50` -- samples with any state exceeding 50 are removed
- **These are applied in `data_collector.py:224-227`** during data collection, not at runtime

### 2.6 Speed Range

- Data collection: v in [2.0, 6.0] m/s (uniform random)
- Normalization scale: v / 5.0

### 2.7 Normalization Constants

| State | Scale | Operation |
|-------|-------|-----------|
| ey | 10.0 | divide |
| epsi | 1.57 | divide |
| v | 5.0 | divide |
| theta | 1.57 | divide |
| theta_dot | 10.0 | divide |
| k | 8.0 | **multiply** (inverted!) |
| delta | 0.785 | divide |
| delta_dot | 3.0 | divide |

**Evidence**: CONFIRMED by `data_collector.py:17-26`, `data_collector.py:39` (k is multiplied, not divided)

---

## 3. State Properties Analysis

### 3.1 Classification Table

| State | Category | Dynamics | Should Predict? | Rationale |
|-------|----------|----------|-----------------|-----------|
| ey | Path error | Yes (integrates epsi) | **YES** | Core tracking state |
| epsi | Path error | Yes (integrates v, delta) | **YES** | Core tracking state |
| v | Vehicle state | Yes (set by rider/controller) | **MAYBE** | v is roughly constant in data; no throttle model |
| theta | Vehicle internal | Yes (gravity + coupling) | **YES** | Core balance state |
| theta_dot | Vehicle internal | Yes (dynamics) | **YES** | Core balance state |
| **k** | **Exogenous** | **NO (constant per episode)** | **NO** | Set at episode start, never changes |
| delta | Vehicle internal | Yes (steering dynamics) | **YES** | Core steering state |
| delta_dot | Vehicle internal | Yes (steering dynamics) | **YES** | Core steering state |

### 3.2 Critical Finding: kappa Should Be an Exogenous Input

**Evidence**: CONFIRMED by code analysis

1. In `data_collector.py:128-135`, `k_ref` is set once per episode via `_random_initial_state()`:
   ```python
   if scenario == 'straight':
       k_ref = self.rng.uniform(-0.02, 0.02)
   ```
2. In `data_collector.py:209`, k is injected as a constant:
   ```python
   states_8d = np.array([full_state_to_path_tracking(xi, k_ref) for xi in x_full])
   ```
3. In `bicycle_dynamics.py:227`:
   ```python
   return np.array([ey, epsi, v, theta, theta_dot, ref_path_curvature, delta, delta_dot])
   ```
4. `ref_path_curvature` is the same constant for ALL time steps within an episode
5. Therefore `Delta_k = k_{t+1} - k_t = 0` always, and the SINDy model learns a trivial `Delta_k = 0` equation for this dimension
6. The SINDy library includes terms like `k*delta`, `k*v`, etc., which provide no predictive value since k is constant

**Conclusion**: k should be reclassified as an **exogenous input** alongside the steering torque, making the system effectively 7D state + 2D input (k, torque). Alternatively, k can be kept as part of the state but the SINDy model should NOT attempt to predict it.

### 3.3 Speed State (v)

In the current data generation, v is set randomly in [2, 6] at episode start and the dynamics model has `dv/dt = 0` (line 160 of `bicycle_dynamics.py`: the time derivative of v is 0). Like k, v is effectively constant per episode.

**However**, v is a genuine physical state that CAN change in a real bicycle (braking, pedaling). The simplified model just does not model speed changes. So v is in a gray area: it is constant in the current data but would be dynamic in a more complete model.

---

## 4. Models and Evaluation

### 4.1 SINDy Model

| Property | sindy_model.npz | sindy_model_improved.npz |
|----------|-----------------|--------------------------|
| Coefficient shape | (55, 8) | (55, 8) |
| Library | Degree-2 polynomial | Degree-2 polynomial |
| Library terms | 55 = 1 + 9 + C(9+1,2) | 55 |
| Features | [8 states, 1 action] | [8 states, 1 action] |
| Method | STLSQ | STLSQ |
| Threshold | Unknown (saved) | Unknown (saved) |

### 4.2 GP Model for 8D

**DOES NOT EXIST**. The `GPMethod` in `methods_classic.py` is hardcoded for 4D:
- Input dimension: 5D (4 state + 1 action, normalized)
- Output dimension: 4D (delta state, normalized)
- Trained on data from `methods_common.py:generate_training_data()` which generates 4D Meijaard data

### 4.3 NN Residual Model for 8D

**DOES NOT EXIST**. The `ResidualNet` in `methods_nn.py` is hardcoded for `state_dim=4`:
```python
class ResidualNet(nn.Module):
    def __init__(self, state_dim: int = 4, action_dim: int = 1, hidden: int = 128):
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden), nn.SiLU(),
            ...
        )
```

All test files (`test_gp_nn_improved.py`, `test_all_improved.py`, `test_all_methods_improved.py`) import from `methods_common.py` and use 4D data exclusively.

### 4.4 Available 8D Model Summary

| Model | Exists | Type | Notes |
|-------|--------|------|-------|
| SINDy (original) | YES | Sparse polynomial | 55x8 coefficients |
| SINDy (improved) | YES | Sparse polynomial | 55x8 coefficients |
| GP baseline | **NO** | - | 4D only |
| GP+NN residual | **NO** | - | 4D only |
| NN residual (ResidualNet) | **NO** | - | 4D only |
| Neural ODE | **NO** | - | 4D only |

### 4.5 Known Instability Issue

In `sindy_env.py:142`:
```python
delta = delta * 0.5  # scale down dynamics for stability
```
The SINDy model requires a 50% damping factor to prevent divergence. This indicates the identified model is significantly over-predicting state changes.

---

## 5. Critical Issues Found

| # | Issue | Severity | Evidence |
|---|-------|----------|----------|
| 1 | k is constant per episode but treated as a state to predict | HIGH | `data_collector.py:209`, `bicycle_dynamics.py:227` |
| 2 | v is constant per episode (no speed model) | MEDIUM | `bicycle_dynamics.py:160` |
| 3 | No GP/NN models exist for 8D route | HIGH | All GP/NN code is 4D-specific |
| 4 | Very small dataset (468 or 2083 samples for 55-term library) | HIGH | Library has 55 terms x 8 outputs = 440 parameters |
| 5 | SINDy model requires 0.5x damping for stability | HIGH | `sindy_env.py:142` |
| 6 | No evaluation framework exists for 8D multi-step rollout | HIGH | No 8D-specific evaluation code found |
| 7 | Fall detection threshold (1.0 rad) is very generous | LOW | ~57 degrees lean angle |

---

## 6. Recommendations

1. **Reclassify k as exogenous input**: Move k from state to input alongside torque. This makes the prediction problem 7D output instead of 8D, removes the trivial `Delta_k = 0` equation, and improves library conditioning.

2. **Add speed dynamics**: Either model speed changes (throttle/braking) or treat v as another exogenous parameter if speed is fixed.

3. **Train GP/NN for 8D**: Port the 4D GP/NN framework to 8D (change input/output dimensions, use 8D normalization constants). This is a straightforward adaptation since the architecture is the same.

4. **Increase data scale**: 2083 samples for a 55x8 system is marginal. At minimum, 10,000+ samples are needed for robust identification.

5. **Implement multi-step evaluation**: Create an 8D-specific evaluation script that tests 1/5/10/20/50/100/500 step rollouts, similar to the 4D evaluation in `test_gp_nn_improved.py`.

---

*Audit completed: 2026-06-25*
*Auditor: Claude Code*
