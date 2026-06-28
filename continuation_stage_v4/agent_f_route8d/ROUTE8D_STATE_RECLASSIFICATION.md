# 8D Route State Reclassification

> **Generated**: 2026-06-25
> **Agent**: F (Route 8D Restructuring Planner)
> **Evidence Levels**: CONFIRMED (code-verified), INFERRED (reasoned from physics), UNKNOWN (unconfirmed)

---

## 1. Current 8D State Vector

```
x = [e_y, e_psi, v, theta, theta_dot, k, delta, delta_dot]
```

Control input: tau (steering torque, 1D, range [-10, 10])

---

## 2. Per-Variable Classification

| Variable | Unit | Category | Dynamic? | Predict? | Evidence | Rationale |
|----------|------|----------|----------|----------|----------|-----------|
| e_y | m | **Path Error State** | Yes (integrates e_psi * v) | YES | CONFIRMED | Pure kinematic: de_y/dt = v * sin(e_psi) |
| e_psi | rad | **Path Error State** | Yes (integrates v*delta/L - v*kappa) | YES | CONFIRMED | Pure kinematic: de_psi/dt = v*delta/L - v*kappa |
| v | m/s | **Vehicle Dynamic State** (or Exogenous) | Constant in data (dv/dt=0, line 160) | MAYBE | CONFIRMED: `bicycle_dynamics.py:160` | Physical state but no throttle model; constant per episode |
| theta | rad | **Vehicle Dynamic State** (Balance) | Yes (gravity + steering coupling) | YES | CONFIRMED | Core balance state; theta_ddot depends on gravity, v, delta |
| theta_dot | rad/s | **Vehicle Dynamic State** (Balance) | Yes (dynamics) | YES | CONFIRMED | Core balance state; paired with theta |
| k | 1/m | **Exogenous Context** | NO (constant per episode) | NO | CONFIRMED: `data_collector.py:128-135,209` | Set at episode start, never changes; Delta_k = 0 always |
| delta | rad | **Vehicle Dynamic State** (Steering) | Yes (steering dynamics) | YES | CONFIRMED | Core steering state; coupled with tau and theta |
| delta_dot | rad/s | **Vehicle Dynamic State** (Steering) | Yes (steering dynamics) | YES | CONFIRMED | Core steering state; paired with delta |
| tau | N*m | **Control Input** | N/A (external) | N/A | CONFIRMED | Steering torque command |

---

## 3. Classification Justification

### 3.1 Path Error States: e_y, e_psi

**Category**: Path Error State

These are purely kinematic variables that describe the bicycle's deviation from the reference path:

```
de_y/dt = v * sin(e_psi)
de_psi/dt = v * tan(delta) / L - v * kappa
```

They depend on v, delta, and kappa but have no mass/inertia associated. They are essential for path-tracking control but are not "physical" dynamic states of the bicycle itself.

**Evidence**: CONFIRMED by `bicycle_dynamics.py:203-227` -- `full_state_to_path_tracking()` computes them from position and heading.

### 3.2 Vehicle Dynamic States: theta, theta_dot, delta, delta_dot

**Category**: Vehicle Dynamic State (Physical)

These represent the actual mechanical degrees of freedom of the bicycle:

- **theta, theta_dot**: Roll (lean) dynamics. The bicycle falls due to gravity; recovery requires steering. Core balance states.
- **delta, delta_dot**: Steering dynamics. The steering angle and its rate are the primary control-relevant physical states.

**Evidence**: CONFIRMED by `bicycle_dynamics.py:131-147` -- the equations of motion compute theta_ddot and delta_ddot from forces and torques.

The SINDy identified model confirms strong coupling:
- `delta` equation: dominated by `v*delta`, `theta`, `delta`, `v*theta`, `v*theta_dot`, `delta_dot`
- `delta_dot` equation: 31 active terms including `v*delta`, `theta`, `delta`, `v*theta`, `v*theta_dot`, `a` (torque)
- `theta` equation: `theta_dot` term (kinematic integration)
- `theta_dot` equation: `v*delta` term (steering-leverage coupling)

### 3.3 Exogenous Context: k (kappa)

**Category**: Exogenous Context (NOT a state)

**Key evidence**:
1. `data_collector.py:128-135`: k is set once at episode start via `_random_initial_state()`
2. `data_collector.py:209`: k is broadcast constant to all timesteps: `states_8d = np.array([full_state_to_path_tracking(xi, k_ref) for xi in x_full])`
3. `bicycle_dynamics.py:227`: k is passed as `ref_path_curvature` -- a fixed parameter
4. `bicycle_dynamics.py:160`: The time derivative of v is 0, and k is not even a dynamic variable -- it is a parameter of the reference path
5. The SINDy model identifies `Delta_k = 0` (0 active terms for k)

**Physical justification**: k is a property of the reference path the bicycle follows, not a state that evolves under bicycle dynamics. It is analogous to road curvature for a car or current velocity for a boat -- an environmental parameter.

**Data contamination**: With k in the state vector, the polynomial library includes degenerate cross-terms (k*ey, k*delta, etc.) that add collinearity without predictive value. This degrades STLSQ performance for the other 7 states.

### 3.4 Speed State: v (Gray Area)

**Category**: Vehicle Dynamic State (but constant in current data)

**Evidence**: `bicycle_dynamics.py:160` shows `dv/dt = 0` -- speed is constant in the current simplified model.

**Why not exogenous?**
- In a real bicycle, v is a physical state that changes with pedaling and braking
- The Whipple model does have speed-dependent coupling (centrifugal effects, steering geometry)
- v is NOT set as a fixed constant like k -- it is initialized randomly in [2,6] and would be dynamic in a more complete model
- The SINDy model shows v participates in coupling terms (v*delta, v*theta, v*theta_dot)

**Recommendation**: Keep v as a state in the model. When collecting new data with a throttle/braking model, v will naturally become dynamic. For now, v varies across episodes and provides useful variation for identification.

---

## 4. Recommended 7D Restructuring

Remove k from the state vector, reclassify as exogenous input:

```
State (7D):  x = [e_y, e_psi, v, theta, theta_dot, delta, delta_dot]
Input (2D):  u = [tau, kappa]
```

**Benefits**:
1. Eliminates trivial `Delta_k = 0` equation (was wasting 1/8 of output dimensions)
2. Reduces polynomial library from 55 terms (9 features) to 45 terms (9 features -- same count but 7 outputs instead of 8, so the Xi matrix is 45x7 instead of 55x8)
3. Removes collinearity from k*state cross-terms
4. Physically correct: kappa is a path parameter, not a dynamic state
5. Better conditioned system: 45 parameters vs 440 (55*8) in the original

**Note**: The feature matrix remains 9-dimensional (7 states + 2 inputs = 9 features), so the library is still 45 terms = 1 + 9 + C(9+1,2). However, the Xi matrix becomes 45x7 (7 outputs to predict) instead of 55x8 (8 outputs). This is a significant reduction in estimation burden.

Wait -- actually the library terms count: with 9 features and degree 2:
- 1 (constant) + 9 (linear) + C(10,2) = 1 + 9 + 45 = 55 terms

So the library is still 55 terms, but Xi is 55x7 instead of 55x8. The reduction is from 440 parameters to 385 parameters (12.5% fewer).

**Actually**: The original 8D system has features = [8 states + 1 action] = 9 features. The proposed 7D system has features = [7 states + 2 inputs] = 9 features. Same library size! But the output dimension drops from 8 to 7, removing the trivial equation.

---

## 5. Comparison: 7D vs 6D (v-fixed)

| Aspect | 7D (v as state) | 6D (v fixed at mean) |
|--------|-----------------|---------------------|
| State dim | 7 | 6 |
| Input dim | 2 | 2 (v becomes input alongside tau, kappa) |
| Library features | 9 (7+2) | 9 (6+3) -- same! |
| Library terms | 55 | 55 -- same! |
| Xi matrix | 55 x 7 = 385 params | 55 x 6 = 330 params |
| Physical fidelity | Captures speed-dependent coupling | Loses speed variation |
| When to use | General case, when v varies | Only if speed is truly fixed |
| **Recommendation** | **PREFERRED** | Only for single-speed analysis |

**Recommendation**: Use 7D. Speed variation across episodes (2-6 m/s) is a key excitation source. Fixing v would discard useful information. The 6D model is only appropriate if the application is strictly single-speed.

---

## 6. Impact on Existing Code

### 6.1 `data_collector.py`
- Change `states_8d` to `states_7d` (remove k from state)
- Add `kappa` to inputs: `inputs_2d = np.hstack([actions, kappa_array])`
- Update `normalize_state()` / `denormalize_state()` for 7D
- Update `NORMALIZATION` dict

### 6.2 `sindy_identification.py`
- Update `STATE_NAMES` to 7 entries
- Update `FEATURE_NAMES` to `['ey','epsi','v','theta','theta_dot','delta','delta_dot','tau','kappa']`
- Update `run_sindy_identification()` signature
- Library construction unchanged (still degree-2 over 9 features)

### 6.3 `world_model.py`
- Update `SINDyWorldModel` for 7D state
- Update normalization factors

### 6.4 `sindy_env.py`
- Update `SINDyBicycleEnv` observation space to 7D
- Update normalization to 7D
- Provide kappa as an observation or constructor parameter

---

## 7. Reclassification Summary

| Variable | Old Category | New Category | Action |
|----------|-------------|--------------|--------|
| e_y | State | State (Path Error) | Keep in state |
| e_psi | State | State (Path Error) | Keep in state |
| v | State | State (Vehicle Dynamic) | Keep in state |
| theta | State | State (Vehicle Dynamic) | Keep in state |
| theta_dot | State | State (Vehicle Dynamic) | Keep in state |
| **k** | **State** | **Exogenous Context** | **Move to input** |
| delta | State | State (Vehicle Dynamic) | Keep in state |
| delta_dot | State | State (Vehicle Dynamic) | Keep in state |
| tau | Input | Input (Control) | Keep as input |
| kappa | -- | Input (Exogenous) | **Add to input** |

---

*Reclassification completed: 2026-06-25*
*Agent F (Route 8D Restructuring Planner)*
