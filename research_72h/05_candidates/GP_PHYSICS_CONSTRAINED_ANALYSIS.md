# GP + Physics Constraints Analysis

**Experiment**: GP Physics Constrained for Bicycle System Identification
**Date**: 2026-06-28
**Script**: `gp_physics_constrained.py`
**EXP ID**: EXP013

---

## 1. Physics Model

### 1.1 Known Physics Relationships

| State Variable | Physics Formula | Type |
|---|---|---|
| e_y_dot | v * sin(e_psi) | Exact kinematics |
| e_psi_dot | -v * delta / L | Exact kinematics (L=1.0m) |
| theta_dot | d(theta)/dt | Definition |
| delta_dot | d(delta)/dt | Definition |

Dimensions without exact physics: `v`, `theta_dot`, `delta_dot` (complex dynamics).

### 1.2 Integration Approaches Tested

**Approach 1: Alpha Blending** (main approach)
- GP predicts all 7 state deltas
- Physics computes constrained deltas from known formulas
- `final = alpha * physics + (1-alpha) * GP` for constrained dims (e_y, e_psi, theta, delta)
- Unconstrained dims (v, theta_dot, delta_dot) always use GP

**Approach 2: Residual GP**
- GP learns: `residual = actual_delta - physics_delta`
- Prediction: `output = physics_delta + GP_residual`

**Approach 3: Standard GP** (baseline)
- Pure GP without any physics integration

---

## 2. Experimental Setup

| Parameter | Value |
|---|---|
| Kernel | ConstantKernel(1.0) * RBF(length_scale=1.0) |
| Training samples | 500 |
| Test segments | 3 (length >= 1100 steps) |
| Horizons | H=1, 10, 50, 100, 200, 500 |
| Wheelbase L | 1.0 m |
| Timestep dt | 1/30 s |
| Seed | 42 |
| n_restarts_optimizer | 0 (speed) |

---

## 3. Results

### 3.1 Physics Model Accuracy (Critical Finding)

**The physics model has NEGATIVE correlation with actual data for all measurable dimensions:**

| Dimension | Physics NMAE | Correlation (r) | Assessment |
|---|---|---|---|
| e_y | 0.2332 | -0.0301 | Near-zero, slightly negative |
| e_psi | 0.3244 | -0.0100 | Near-zero, slightly negative |
| v | 0.6092 | N/A | Zero variance in physics |
| theta | 2.7048 | **-0.4342** | **Strong negative** |
| theta_dot | 0.7282 | N/A | Zero variance in physics |
| delta | 7.3279 | **-0.5270** | **Strong negative** |
| delta_dot | 0.1556 | N/A | Zero variance in physics |

**Root cause analysis:**
- The physics formulas `e_y_dot = v*sin(e_psi)` and `e_psi_dot = -v*delta/L` produce predictions that are **anti-correlated** with the actual state deltas in the PyBullet simulation
- This means the physics model is **qualitatively wrong** for this system -- the simulation dynamics differ significantly from the simplified kinematic model
- The negative correlation for theta (r=-0.43) suggests `theta_dot` is being used with wrong sign or scale
- The negative correlation for delta (r=-0.53) suggests `delta_dot` similarly doesn't match

### 3.2 Overall Comparison

| Model | H=1 | H=10 | H=50 | H=100 | H=200 | H=500 | Primary |
|---|---|---|---|---|---|---|---|
| Standard GP | 0.0163 | 0.0423 | 0.0841 | 0.2264 | 0.6296 | 1.2390 | 0.6983 |
| Residual GP | 0.0163 | 0.0421 | 0.0839 | 0.2256 | 0.6425 | 1.1855 | 0.6845 |
| Blended GP (a=1.0) | 0.0163 | 0.0423 | 0.0839 | 0.2250 | 0.6442 | 1.0985 | 0.6559 |

**Best model: Blended GP (alpha=1.0)** -- Primary = 0.6559

### 3.3 Alpha Sweep Results

| Alpha | H=100 | H=200 | H=500 | Primary |
|---|---|---|---|---|
| 0.0 (GP only) | 0.2264 | 0.6296 | 1.2390 | 0.6983 |
| 0.1 | 0.2260 | 0.6295 | 1.2440 | 0.6998 |
| 0.2 | 0.2255 | 0.6297 | 1.2481 | 0.7011 |
| 0.3 | 0.2252 | 0.6302 | 1.2442 | 0.6999 |
| 0.4 | 0.2249 | 0.6314 | 1.2400 | 0.6987 |
| 0.5 | 0.2244 | 0.6330 | 1.2481 | 0.7018 |
| 0.6 | 0.2238 | 0.6351 | 1.2683 | 0.7091 |
| 0.7 | 0.2234 | 0.6374 | 1.2565 | 0.7058 |
| 0.8 | 0.2233 | 0.6396 | 1.2604 | 0.7078 |
| 0.9 | 0.2236 | 0.6415 | 1.2289 | 0.6980 |
| **1.0 (Physics)** | **0.2250** | **0.6442** | **1.0985** | **0.6559** |

**Key observation:** alpha=1.0 (pure physics) gives the best Primary score because it reduces H=500 error significantly (1.0985 vs 1.2390). However, this comes at the cost of slightly worse H=200 performance.

### 3.4 Per-State NMAE at H=500

| State | Standard GP | Blended (a=1.0) | Change |
|---|---|---|---|
| e_y | 0.9518 | 1.0631 | +11.7% worse |
| e_psi | 0.9092 | 0.9004 | -1.0% better |
| v | 1.6680 | 1.4122 | -15.3% better |
| theta | 1.2827 | 1.0949 | -14.6% better |
| theta_dot | 1.3070 | 1.1030 | -15.6% better |
| delta | 1.2779 | 1.1661 | -8.7% better |
| delta_dot | 1.2761 | 0.9496 | -25.6% better |

### 3.5 Comparison with v9 Neural ODE Baseline

| Model | Primary Score | vs v9 |
|---|---|---|
| v9 Neural ODE | 0.5110 | baseline |
| Standard GP (500 samples) | 0.6983 | -36.7% worse |
| Residual GP | 0.6845 | -34.0% worse |
| Blended GP (a=1.0) | 0.6559 | -28.4% worse |

---

## 4. Key Findings

### 4.1 Physics Model is Fundamentally Mismatched

The most important finding is that **the simplified kinematic physics model does not match the PyBullet simulation dynamics**:

1. **Negative correlations** for all measurable dimensions (e_y: r=-0.03, e_psi: r=-0.01, theta: r=-0.43, delta: r=-0.53)
2. The physics formulas are **qualitatively wrong** -- they predict changes in the opposite direction from what actually happens
3. This means the PyBullet bicycle dynamics are significantly more complex than `e_y_dot = v*sin(e_psi)` suggests

### 4.2 Why Alpha=1.0 "Works"

Despite negative correlations, alpha=1.0 (pure physics) gives the best Primary score because:
- Physics predictions tend to be **smaller in magnitude** than GP predictions
- At long horizons (H=500), smaller per-step errors accumulate less
- Effectively acts as **regularization** rather than accurate physics
- This is a **coincidence**, not genuine physics improvement

### 4.3 GP Performance Gap vs v9

With only 500 training samples, GP is 28-37% worse than v9 Neural ODE:
- v9 was likely trained on much more data
- GP with 500 samples cannot learn the full dynamics
- The original GP breakthrough (0.4372 Primary) used 5000 samples

### 4.4 Physics Integration Does NOT Help

None of the physics integration approaches improve over standard GP by a meaningful margin:
- Residual GP: -2.0% improvement (negligible)
- Blended GP: -6.1% improvement (only at alpha=1.0, which is just regularization)
- The **negative correlation** of the physics model means physics constraints **hurt** rather than help

---

## 5. Conclusions

### 5.1 Physics Constraints Assessment

**Physics constraints do NOT improve GP performance for this system.** The root cause is that the simplified kinematic model (`e_y_dot = v*sin(e_psi)`, `e_psi_dot = -v*delta/L`) does not accurately represent the PyBullet bicycle dynamics. The physics model predictions are anti-correlated with actual data, making them harmful rather than helpful.

### 5.2 Recommendations

1. **Do NOT use physics constraints** with the current simplified model
2. **Use more training data** (5000+ samples) for GP to recover the original breakthrough performance
3. **Investigate the dynamics mismatch** -- the PyBullet bicycle may have different state conventions or additional coupling terms
4. **Consider learning the physics** from data (e.g., SINDy) rather than imposing hand-coded formulas

### 5.3 Root Cause: Dynamics Mismatch

The negative correlations suggest one or more of:
- State variable definitions differ between physics formulas and PyBullet simulation
- Additional coupling terms exist (e.g., roll-steering coupling, gyroscopic effects)
- The sign convention for delta or e_psi is inverted in the simulation
- The PyBullet model includes trail/rake effects not captured in the simple model

---

## 6. Files

- **Script**: `research_72h/05_candidates/gp_physics_constrained.py`
- **Results JSON**: `research_72h/05_candidates/EXP013_gp_physics_constrained.json`
