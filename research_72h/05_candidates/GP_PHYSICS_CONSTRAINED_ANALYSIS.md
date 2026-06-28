# GP + Physics Constraints Analysis

**Experiment**: GP Physics Constrained for Bicycle System Identification
**Date**: 2026-06-28
**Script**: `gp_physics_constrained.py`

---

## 1. Physics Model

### 1.1 Known Physics Relationships

The bicycle model has two exact kinematic relationships:

| State Variable | Physics Formula | Type |
|---|---|---|
| e_y_dot | v * sin(e_psi) | Exact kinematics |
| e_psi_dot | -v * delta / L | Exact kinematics (L=wheelbase) |
| theta_dot | d(theta)/dt | Definition |
| delta_dot | d(delta)/dt | Definition |

Dimensions without exact physics models: `v`, `theta_dot`, `delta_dot` (involve complex dynamics like drag, gravity, spring/damping).

### 1.2 Integration Approaches

Three approaches to integrate physics into GP:

**Approach 1: Alpha Blending**
- GP predicts all 7 state deltas
- Physics computes e_y_dot, e_psi_dot, theta_dot, delta_dot from known formulas
- Final = alpha * physics + (1-alpha) * GP for constrained dimensions
- Unconstrained dimensions use GP only

**Approach 2: Physics-Augmented Features**
- Add 2 physics-derived features to GP input: `v*sin(e_psi)` and `-v*delta/L`
- GP learns to use these features alongside raw state/action
- No post-processing needed

**Approach 3: Residual GP**
- GP learns residual = actual_delta - physics_delta
- At prediction: output = physics_delta + GP_residual
- Physics provides the prior, GP learns the correction

---

## 2. Experimental Setup

| Parameter | Value |
|---|---|
| Kernel | ConstantKernel(1.0) * RBF(length_scale=1.0) |
| Training samples | 5,000 |
| Test segments | 5 (length >= 1100 steps) |
| Horizons | H=1, 10, 50, 100, 200, 500 |
| Wheelbase L | 1.0 m |
| Timestep dt | 1/30 s |
| Seed | 42 |

---

## 3. Results

*(To be filled after experiment completes)*

### 3.1 Physics Model Accuracy

The physics model's NMAE on training data shows how well the known formulas predict each dimension.

### 3.2 Overall Comparison

| Model | H=1 | H=10 | H=50 | H=100 | H=200 | H=500 | Primary |
|---|---|---|---|---|---|---|---|
| Standard GP | - | - | - | - | - | - | - |
| Physics-Augmented GP | - | - | - | - | - | - | - |
| Residual GP | - | - | - | - | - | - | - |
| Blended GP (best alpha) | - | - | - | - | - | - | - |

### 3.3 Alpha Sweep

| Alpha | H=100 | H=200 | H=500 | Primary |
|---|---|---|---|---|
| 0.0 (GP only) | - | - | - | - |
| 0.1 | - | - | - | - |
| 0.2 | - | - | - | - |
| 0.3 | - | - | - | - |
| 0.4 | - | - | - | - |
| 0.5 | - | - | - | - |
| 0.6 | - | - | - | - |
| 0.7 | - | - | - | - |
| 0.8 | - | - | - | - |
| 0.9 | - | - | - | - |
| 1.0 (Physics only) | - | - | - | - |

### 3.4 Per-State Analysis

*(Best model vs Standard GP per-state NMAE at H=100, 200, 500)*

### 3.5 Comparison with v9 Neural ODE

| Model | Primary Score | vs v9 |
|---|---|---|
| v9 Neural ODE | 0.5110 | baseline |
| Standard GP | - | - |
| Best Physics-Constrained GP | - | - |

---

## 4. Key Findings

*(To be filled after analysis)*

### 4.1 Does Physics Help?

### 4.2 Which Approach Works Best?

### 4.3 Optimal Alpha

### 4.4 Which Dimensions Benefit Most?

---

## 5. Conclusions

*(To be filled after analysis)*

---

## 6. Files

- **Script**: `research_72h/05_candidates/gp_physics_constrained.py`
- **Results JSON**: `research_72h/05_candidates/EXP013_gp_physics_constrained.json`
