# 8D Route Layered Architecture

> **Generated**: 2026-06-25
> **Agent**: F (Route 8D Restructuring Planner)
> **Design Philosophy**: Decompose the monolithic SINDy model into physically motivated layers, each with identifiable structure and clear interpretation.

---

## 1. Overview: Four-Layer Architecture

The bicycle path-tracking system can be decomposed into four physically distinct layers, each governing a subset of the state variables. This decomposition enables:

1. **Per-layer SINDy identification** with smaller, better-conditioned systems
2. **Physical consistency** by encoding known kinematic relationships exactly
3. **Selective coupling** where layers interact through explicit terms
4. **Incremental complexity** -- start with simple models, add coupling as data allows

```
Layer 1: Path Kinematics        [e_y, e_psi]
Layer 2: Balance Dynamics       [theta, theta_dot]
Layer 3: Steering Actuator      [delta, delta_dot]
Layer 4: Coupling + Residuals   (cross-layer terms)
```

Exogenous inputs: `v`, `kappa` (context), `tau` (control)

---

## 2. Layer 1: Path Kinematics

### 2.1 States
```
[e_y, e_psi]
```

### 2.2 Known Equations (Exact)
These are kinematic relationships that are known exactly from bicycle geometry:

```
de_y/dt = v * sin(e_psi)
de_psi/dt = v * tan(delta) / L - v * kappa
```

Discrete time (dt = 1/30):
```
e_y_{t+1} = e_y_t + v_t * sin(e_psi_t) * dt
e_psi_{t+1} = e_psi_t + (v_t * tan(delta_t) / L - v_t * kappa_t) * dt
```

### 2.3 Strategy
- **Encode these equations analytically** in the model (do NOT learn them from data)
- This removes 2 output dimensions from the SINDy identification problem
- The only unknown is the wheelbase L, which is known (1.02 m from `BicycleParams`)
- Small-angle approximation: sin(e_psi) ~ e_psi, tan(delta) ~ delta for small angles

### 2.4 Data Requirement
- Zero additional data needed -- these are geometric identities
- Validation only: compare analytical prediction vs data residuals

### 2.5 Confidence: CONFIRMED
The kinematic equations are exact for the bicycle model used in data generation.

---

## 3. Layer 2: Balance Dynamics

### 3.1 States
```
[theta, theta_dot]
```

### 3.2 Known Structure (from Whipple model)
The roll dynamics have a known physical structure:

```
I_roll * theta_ddot = m*g*h*sin(theta) + centrifugal_coupling + steering_leverage
```

Discrete:
```
theta_{t+1} = theta_t + theta_dot_t * dt
theta_dot_{t+1} = theta_dot_t + f_balance(theta_t, theta_dot_t, delta_t, delta_dot_t, v_t, kappa_t) * dt
```

### 3.3 SINDy Identification for theta_dot
The theta equation is purely kinematic (theta integrates theta_dot). Only theta_dot needs identification:

```
theta_dot equation features: [theta, theta_dot, delta, delta_dot, v, kappa, tau]
```

**Expected active terms** (from existing SINDy model and physics):
- `theta` (gravity restoring torque)
- `theta_dot` (damping)
- `v*delta` (steering-leverage coupling, dominant term)
- `v*theta_dot` (speed-dependent damping)
- `delta` (direct steering effect)

### 3.4 SINDy Library Size
- Input features for this layer: 7 (theta, theta_dot, delta, delta_dot, v, kappa, tau)
- Library terms: 1 + 7 + C(8,2) = 1 + 7 + 28 = 36
- Output dimensions: 1 (theta_dot equation only; theta is kinematic integration)

### 3.5 Strategy
1. Encode `theta_{t+1} = theta_t + theta_dot_t * dt` analytically
2. Identify `theta_dot` equation via SINDy with 7 input features
3. This is a 36x1 system (36 library terms, 1 output) -- very well-conditioned

### 3.6 Existing Model Validation
The current improved SINDy model identifies:
- `Delta_theta = 0.214 * theta_dot` (kinematic integration, correct)
- `Delta_theta_dot = 0.105 * v*delta` (steering-leverage coupling, physically correct)

These match the expected physics. The 0.5x damping factor in `sindy_env.py:142` suggests the model over-predicts theta_dot changes, likely because:
1. The model was identified on the full 8D system with k contamination
2. Damping terms were thresholded out
3. Cross-layer coupling with delta was partially absorbed

### 3.7 Confidence: CONFIRMED (structure) / INFERRED (SINDy coefficients)

---

## 4. Layer 3: Steering Actuator

### 4.1 States
```
[delta, delta_dot]
```

### 4.2 Known Structure
The steering dynamics are governed by:

```
I_steer * delta_ddot = tau - m*g*c*sin(theta)/L + m*v^2*c*delta/L^2 - damping*delta_dot
```

Discrete:
```
delta_{t+1} = delta_t + delta_dot_t * dt
delta_dot_{t+1} = delta_dot_t + f_steer(delta_t, delta_dot_t, theta_t, theta_dot_t, v_t, kappa_t, tau_t) * dt
```

### 4.3 SINDy Identification for delta_dot
The delta equation is purely kinematic. Only delta_dot needs identification:

```
delta_dot equation features: [delta, delta_dot, theta, theta_dot, v, kappa, tau]
```

**Expected active terms** (from existing SINDy model and physics):
- `tau` (steering torque -- the primary input)
- `delta` (steering stiffness)
- `delta_dot` (steering damping)
- `theta` (gravity coupling -- the self-steering effect)
- `v*delta` (speed-dependent steering stiffness)
- `v*delta_dot` (speed-dependent steering damping)

### 4.4 SINDy Library Size
- Input features for this layer: 7 (delta, delta_dot, theta, theta_dot, v, kappa, tau)
- Library terms: 1 + 7 + C(8,2) = 36
- Output dimensions: 1 (delta_dot equation only)

### 4.5 Existing Model Validation
The current improved SINDy model identifies 31 active terms for delta_dot. The top terms are:
- `v*delta` (20.68) -- dominant, physically correct
- `theta` (-8.19) -- gravity coupling
- `delta` (-6.41) -- steering stiffness
- `v*theta` (-4.32) -- speed-gravity coupling
- `v*theta_dot` (-1.33) -- speed-damping coupling
- `tau` (0.28) -- steering torque (relatively small coefficient, but this is because the coefficient is in normalized space and action_scale = 1.41)

The 31 active terms suggest over-fitting. With 36 total library terms, 31 is not sparse. This is likely due to:
1. Small dataset (2083 samples)
2. k contamination adding collinear terms
3. High threshold not applied aggressively enough

### 4.6 Strategy
1. Encode `delta_{t+1} = delta_t + delta_dot_t * dt` analytically
2. Identify `delta_dot` equation via SINDy with 7 input features
3. Expect 5-10 active terms after proper identification with more data

### 4.7 Confidence: CONFIRMED (structure) / INFERRED (SINDy coefficients)

---

## 5. Layer 4: Coupling and Residuals

### 5.1 Purpose
Capture cross-layer interactions that the isolated Layer 2 and Layer 3 models miss.

### 5.2 Identified Coupling Terms

From the existing SINDy model, the dominant cross-layer couplings are:

| Coupling | Source | Target | Type | Strength |
|----------|--------|--------|------|----------|
| v*delta | Steering (Layer 3) | Roll (Layer 2) | Steering leverage | STRONG |
| theta*delta | Roll (Layer 2) | Steering (Layer 3) | Gravity-steering | MODERATE |
| v*theta | Roll (Layer 2) | Steering (Layer 3) | Speed-gravity | MODERATE |
| v*theta_dot | Roll rate (Layer 2) | Steering (Layer 3) | Gyroscopic | MODERATE |
| ey^2 | Path error (Layer 1) | Steering (Layer 3) | Nonlinear path | WEAK |
| epsi*delta | Path error (Layer 1) | Steering (Layer 3) | Heading-steering | WEAK |
| ey*v | Path error (Layer 1) | Steering (Layer 3) | Path-speed | WEAK |

### 5.3 Strategy
After identifying Layer 2 and Layer 3 independently:

1. Compute residuals: `residual_steer = delta_dot_true - f_steer_predicted`
2. Identify residual dynamics using cross-layer features
3. If residuals are small (<5% of signal), skip residual layer
4. If residuals are significant, add cross-layer SINDy terms

### 5.4 Confidence: INFERRED (coupling strengths from existing model)

---

## 6. Complete Architecture Summary

### 6.1 Hybrid Analytical + SINDy Model

```
Input:  u = [tau, kappa]
State:  x = [e_y, e_psi, v, theta, theta_dot, delta, delta_dot]

Step 1: Path Kinematics (ANALYTICAL)
  e_y_{t+1} = e_y_t + v_t * sin(e_psi_t) * dt
  e_psi_{t+1} = e_psi_t + (v_t * tan(delta_t) / L - v_t * kappa_t) * dt

Step 2: Balance Dynamics (SINDy for theta_dot)
  theta_{t+1} = theta_t + theta_dot_t * dt               [ANALYTICAL]
  theta_dot_{t+1} = theta_dot_t + SINDy_balance(...) * dt [LEARNED]

Step 3: Steering Actuator (SINDy for delta_dot)
  delta_{t+1} = delta_t + delta_dot_t * dt                [ANALYTICAL]
  delta_dot_{t+1} = delta_dot_t + SINDy_steer(...) * dt   [LEARNED]

Step 4: Speed (CONSTANT or EXOGENOUS)
  v_{t+1} = v_t  [CONSTANT in current model]
  (or: v_{t+1} = f_throttle(v_t, throttle_cmd) if available)

Step 5: Coupling (OPTIONAL SINDy residual)
  Apply cross-layer correction if residuals > threshold
```

### 6.2 Effective Model Dimensionality

| Component | Analytical | SINDy-learned | Total |
|-----------|-----------|---------------|-------|
| Path kinematics | 2 (e_y, e_psi) | 0 | 2 |
| Balance | 1 (theta) | 1 (theta_dot) | 2 |
| Steering | 1 (delta) | 1 (delta_dot) | 2 |
| Speed | 1 (v) | 0 | 1 |
| **Total** | **5** | **2** | **7** |

The SINDy identification problem is reduced from 8 outputs to **2 outputs** (theta_dot and delta_dot). This is a massive reduction:

- Original: 55 x 8 = 440 parameters
- Layered: 36 x 2 = 72 parameters (for two independent SINDy models)
- Reduction: **84% fewer parameters**

### 6.3 Advantages

1. **Physical consistency**: Kinematic relationships are exact, not approximated
2. **Better conditioning**: Each SINDy model has only 36 library terms and 1 output
3. **Interpretability**: Each equation has clear physical meaning
4. **Extensibility**: Can add throttle model, rider model, etc. as new layers
5. **Reduced data need**: 72 parameters vs 440 -- needs ~6x fewer samples

---

## 7. Comparison: Monolithic vs Layered

| Aspect | Monolithic 7D SINDy | Layered (5+2) |
|--------|---------------------|---------------|
| SINDy outputs | 7 | 2 |
| Library terms per model | 55 | 36 x 2 = 72 total |
| Total parameters | 385 | 72 |
| Data requirement | ~4000+ samples | ~700+ samples |
| Physical consistency | Approximate | Exact for kinematics |
| Interpretability | Low (black-box equations) | High (each layer = physics) |
| Known bug: 0.5x damping | Likely caused by kinematic error | Eliminates kinematic error |
| Validation difficulty | Hard to diagnose | Easy to isolate per layer |

---

## 8. Implementation Priority

| Priority | Task | Effort | Impact |
|----------|------|--------|--------|
| P0 | Remove k from state, make exogenous | LOW | HIGH |
| P1 | Encode path kinematics analytically | LOW | HIGH |
| P2 | Encode theta integration analytically | LOW | MEDIUM |
| P3 | Encode delta integration analytically | LOW | MEDIUM |
| P4 | SINDy for theta_dot only | MEDIUM | HIGH |
| P5 | SINDy for delta_dot only | MEDIUM | HIGH |
| P6 | Validate per-layer single-step | LOW | HIGH |
| P7 | Validate 1/5/10/50 step rollout | MEDIUM | HIGH |
| P8 | Add cross-layer residual if needed | HIGH | MEDIUM |
| P9 | Collect more data if needed | MEDIUM | MEDIUM |

---

## 9. Open Questions

| # | Question | Impact | Confidence |
|---|----------|--------|------------|
| 1 | Should v be constant or dynamic? | HIGH -- affects model structure | INFERRED |
| 2 | Is the theta equation purely kinematic? | MEDIUM -- confirmed in current model | CONFIRMED |
| 3 | How many cross-layer terms are needed? | MEDIUM -- data-dependent | UNKNOWN |
| 4 | Does the layered model eliminate the 0.5x damping? | HIGH -- key validation | UNKNOWN |
| 5 | What is the minimum data per layer? | MEDIUM -- affects collection plan | UNKNOWN |

---

*Architecture designed: 2026-06-25*
*Agent F (Route 8D Restructuring Planner)*
