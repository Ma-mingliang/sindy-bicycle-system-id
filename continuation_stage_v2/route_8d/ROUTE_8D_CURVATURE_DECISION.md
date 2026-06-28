# 8D Route -- Curvature (kappa) Decision Document

> **Generated**: 2026-06-25
> **Decision**: kappa should be reclassified from state to exogenous input

---

## 1. Current Treatment

kappa (k) is currently the 6th element of the 8D state vector:
```
x = [ey, epsi, v, theta, theta_dot, k, delta, delta_dot]
```

The SINDy model predicts the 8D state transition:
```
x_{t+1} = x_t + f_SINDy(x_t, a_t)
```

This means the model attempts to predict `Delta_k = k_{t+1} - k_t`.

---

## 2. Evidence That k Is Not a Dynamic State

### 2.1 Code Evidence (CONFIRMED)

**`data_collector.py:128-135`** -- k is set once per episode:
```python
if scenario == 'straight':
    k_ref = self.rng.uniform(-0.02, 0.02)
elif scenario == 'curved':
    k_ref = self.rng.uniform(-0.1, 0.1)
elif scenario == 'sharp':
    k_ref = self.rng.uniform(-0.3, 0.3)
```

**`data_collector.py:209`** -- k is broadcast to all timesteps:
```python
states_8d = np.array([full_state_to_path_tracking(xi, k_ref) for xi in x_full])
```

**`bicycle_dynamics.py:227`** -- k is a constant parameter:
```python
return np.array([ey, epsi, v, theta, theta_dot, ref_path_curvature, delta, delta_dot])
```

### 2.2 Mathematical Consequence

Since k is constant within each episode:
```
Delta_k = k_{t+1} - k_t = 0  (always)
```

The SINDy identification will find:
```
Delta_k = 0 (or near-zero coefficients)
```

This wastes one column of the Xi coefficient matrix and adds noise to the sparse regression for the other 7 states (because polynomial library terms involving k will have degenerate columns).

### 2.3 Library Contamination

With k in the state, the polynomial library (degree 2) includes cross-terms like:
- `k * ey`, `k * epsi`, `k * v`, `k * theta`, `k * theta_dot`, `k * delta`, `k * delta_dot`
- `k^2`

Since k is constant within each episode but varies across episodes, these terms behave like scaled versions of the non-k terms. They add collinearity to the library matrix, which degrades STLSQ's ability to identify the correct sparse structure for the other 7 states.

---

## 3. Impact Analysis

### 3.1 Current (k as state)

| Aspect | Impact |
|--------|--------|
| Output dimension | 8 (wastes 1 on trivial zero) |
| Library terms | 55 = 1 + 9 + C(10,2) |
| Cross-terms with k | 9 terms (k*each_of_9_features) |
| Collinearity | High (k is near-constant within episodes) |
| Physical accuracy | N/A (Delta_k = 0 is trivially correct) |

### 3.2 Proposed (k as exogenous input)

| Aspect | Impact |
|--------|--------|
| Output dimension | **7** (all non-trivial) |
| Input dimension | **2** (torque + k) |
| Library terms | 36 = 1 + 9 + C(9,2) [with 7 state + 2 input features] |
| Cross-terms with k | k now interacts with states as an input, which is physically correct: curvature affects how state evolves |
| Collinearity | Reduced (k is no longer an output to predict) |

---

## 4. Physical Justification

In real bicycle path tracking:
- **kappa is determined by the reference path**, not by the bicycle dynamics
- The bicycle's dynamics determine how ey, epsi, theta, delta evolve given a fixed curvature
- Curvature is an **environment parameter** that the controller must handle, not a state that evolves

This is analogous to:
- A car on a banked road: road angle is an input, not a state
- A boat in a current: current velocity is an input, not a state
- A robot on a slope: slope angle is an input, not a state

---

## 5. Recommended Action

### 5.1 Short-term (within current framework)

Keep k in the 8D state but **mask** the k output during SINDy identification:
- Set the k column of the target matrix to zero (or exclude it from loss)
- This prevents the library from fitting noise to the trivial `Delta_k = 0`
- Reduces effective output dimension from 8 to 7

### 5.2 Long-term (recommended)

Restructure the state-space as:
```
State (7D): [ey, epsi, v, theta, theta_dot, delta, delta_dot]
Input (2D): [torque, k]
```

This:
1. Eliminates the trivial equation
2. Reduces library from 55 to 36 terms (fewer parameters to estimate)
3. Properly models curvature as an environmental input that affects state evolution
4. Reduces collinearity in the library matrix
5. Is physically more meaningful

### 5.3 Implementation

In `data_collector.py`, change the state representation:
```python
# Before (8D state + 1D action)
states_8d = [ey, epsi, v, theta, theta_dot, k, delta, delta_dot]
action = torque

# After (7D state + 2D input)
states_7d = [ey, epsi, v, theta, theta_dot, delta, delta_dot]
inputs_2d = [torque, k_ref]
```

In `sindy_identification.py`, update `build_feature_matrix`:
```python
# Before
X = np.hstack([states, actions])  # shape (n, 9)

# After
X = np.hstack([states_7d, inputs_2d])  # shape (n, 9) -- same size but different semantics
```

The polynomial library construction remains the same (degree 2 over 9 features = 55 terms), but the coefficient matrix Xi will be (55, 7) instead of (55, 8).

---

## 6. Decision Summary

| Criterion | k as State (current) | k as Input (proposed) |
|-----------|---------------------|----------------------|
| Physical accuracy | Trivial (Delta_k = 0) | Correct (k affects dynamics) |
| Library efficiency | 55 terms, 8 outputs | 55 terms, 7 outputs |
| Collinearity | High | Low |
| Data requirement | Higher (8 outputs) | Lower (7 outputs) |
| Model interpretability | Low (one trivial eq.) | High (all equations meaningful) |
| Complexity | No change | No change |

**DECISION: Reclassify kappa as exogenous input.**

This is the single highest-impact improvement for the 8D route. It will:
- Remove 1 wasted output dimension
- Reduce library collinearity
- Improve sparse regression quality for all other states
- Make the model physically meaningful

---

*Decision document generated: 2026-06-25*
*Auditor: Claude Code*
